package weka.classifiers.rapids;

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Copyright 2015 University of Waikato
 */


import junit.framework.Test;
import junit.framework.TestSuite;
import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;


/**
 * Tests the CuML LearnClassifier
 *
 * @author Justin Liu (justin{[dot]}l{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version $Revision: $
 */
public class CuMLClassifierTest extends AbstractClassifierTest {
    public CuMLClassifierTest(String name) {
        super(name);
    }

    public Classifier getClassifier() {
        return new CuMLClassifier();
    }

    public static Test suite() {
        return new TestSuite(CuMLClassifierTest.class);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(suite());
    }

    @Override
    protected boolean canPredict(int type) {
        return true;
    }
}
